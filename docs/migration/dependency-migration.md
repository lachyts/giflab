# Dependency Migration Guide: Phases 1-4

---
**Document Version:** 1.0  
**Scope:** Dependency management for Critical Code Review Issues Resolution  
**Migration Type:** 🟡 MODERATE - Some new dependencies, mostly optional  
**Last Updated:** 2025-01-11  
---

## Overview

This guide covers dependency installation, management, and troubleshooting for the Phases 1-4 architectural improvements. The project uses Poetry for all dependency management and maintains careful separation between required and optional dependencies.

**🎯 Key Principle:** GifLab core functionality works with minimal dependencies. Enhanced features require additional optional dependencies.

---

## Dependency Architecture

### **Dependency Categories**

#### **1. Core Dependencies** (Required)
| Package | Version | Purpose | Status |
|---------|---------|---------|---------|
| `click` | ^8.0.0 | CLI framework | Existing |
| `numpy` | ^1.20.0 | Numerical operations | Existing |
| `opencv-python` | ^4.5.0 | Image processing | Existing |
| `rich` | ^13.7.0 | CLI formatting | **NEW - Required** |

#### **2. Processing Dependencies** (Core Functionality)
| Package | Version | Purpose | Status |
|---------|---------|---------|---------|
| `Pillow` | ^8.0.0 | Image handling | Existing |
| `pytest` | ^6.0.0 | Testing framework | Existing |

#### **3. Monitoring Dependencies** (Optional)
| Package | Version | Purpose | Status |
|---------|---------|---------|---------|
| `psutil` | Latest | Memory monitoring | **NEW - Optional** |

#### **4. Visualization Dependencies** (Optional) 
| Package | Version | Purpose | Status |
|---------|---------|---------|---------|
| `matplotlib` | Latest | Basic plotting | **Enhanced - Optional** |
| `seaborn` | Latest | Statistical visualization | **NEW - Optional** |
| `plotly` | Latest | Interactive plotting | **NEW - Optional** |

#### **5. ML Dependencies** (Optional)
| Package | Version | Purpose | Status |
|---------|---------|---------|---------|
| `torch` | Latest | PyTorch framework | Existing - Optional |
| `lpips` | Latest | Perceptual metrics | Existing - Optional |

---

## Pre-Migration Assessment

### **Current Dependency Check**
```bash
# Check current Poetry environment
poetry --version
# Expected: Poetry version 1.1.0+

# Check current Python version  
poetry run python --version
# Expected: Python 3.8+

# Assess current dependencies
poetry show --tree
```

### **Identify Missing Dependencies**
```bash
# Test current critical imports
poetry run python -c "
import sys
missing = []

# Test core dependencies
try: import rich
except ImportError: missing.append('rich (REQUIRED)')

try: import psutil  
except ImportError: missing.append('psutil (optional - memory monitoring)')

try: import matplotlib
except ImportError: missing.append('matplotlib (optional - visualization)')

try: import seaborn
except ImportError: missing.append('seaborn (optional - advanced plots)')

try: import plotly
except ImportError: missing.append('plotly (optional - interactive plots)')

if missing:
    print('Missing dependencies:')
    for dep in missing: print(f'  - {dep}')
else:
    print('✅ All dependencies available')
"
```

### **Environment Assessment**
```bash
# Check Poetry lock file status
poetry check

# Verify virtual environment
poetry env info

# Check for dependency conflicts
poetry install --dry-run
```

---

## Migration Procedures

### **Phase 1: Required Dependencies**

#### **Step 1.1: Update Core Dependencies**
```bash
# The rich library should already be in pyproject.toml
# Verify it's listed:
grep "rich" pyproject.toml
# Should show: rich = "^13.7.0" or similar

# Install/update all dependencies
poetry install

# Verify rich installation
poetry run python -c "
from rich.console import Console
from rich.table import Table
console = Console()
console.print('✅ Rich library working')
"
```

#### **Step 1.2: Validate Core Imports**
```bash
# Test critical CLI imports
poetry run python -c "
try:
    from giflab.cli import metrics, cache
    print('✅ Core CLI imports successful')
except ImportError as e:
    print(f'❌ CLI import error: {e}')
    exit(1)

try:
    from giflab.metrics import calculate_comprehensive_metrics
    print('✅ Core metrics imports successful') 
except ImportError as e:
    print(f'❌ Metrics import error: {e}')
    exit(1)
"
```

#### **Step 1.3: CLI Functionality Test**
```bash
# Test enhanced CLI formatting
poetry run python -m giflab --help
# Should display with rich formatting

# Test dependency checking (new functionality)
poetry run python -m giflab deps status
# Should show dependency status overview
```

**✅ Phase 1 Complete When:**
- Rich library installed and working
- CLI commands display with enhanced formatting
- All core imports successful

### **Phase 2: Memory Monitoring Dependencies**

#### **Step 2.1: Install psutil (Optional)**
```bash
# Install psutil for memory monitoring
poetry add psutil

# Verify installation
poetry run python -c "
import psutil
print(f'✅ psutil version: {psutil.__version__}')

# Test basic memory monitoring
memory = psutil.virtual_memory()
print(f'System memory: {memory.total // (1024**3)}GB total')
print(f'Available: {memory.available // (1024**3)}GB ({memory.percent:.1f}% used)')
"
```

#### **Step 2.2: Test Memory Monitoring Integration**
```bash
# Test memory monitoring system
poetry run python -c "
from giflab.monitoring import SystemMemoryMonitor
monitor = SystemMemoryMonitor()
stats = monitor.get_memory_stats()
print(f'✅ Memory monitoring active')
print(f'Available memory: {stats.available_gb:.1f}GB')
print(f'Memory pressure: {monitor.get_pressure_level().name}')
"
```

#### **Step 2.3: Validate CLI Integration**
```bash
# Test enhanced dependency reporting
poetry run python -m giflab deps check --verbose
# Should show memory monitoring status

# Test memory status reporting
poetry run python -m giflab deps status
# Should include memory monitoring information
```

**✅ Phase 2 Complete When:**
- psutil installed and working
- Memory monitoring system operational
- CLI commands report memory status

### **Phase 3: Visualization Dependencies (Optional)**

#### **Step 3.1: Install Visualization Libraries**
```bash
# Install matplotlib (most commonly used)
poetry add matplotlib

# Install seaborn (statistical plotting)
poetry add seaborn

# Install plotly (interactive plots) 
poetry add plotly

# Verify installations
poetry run python -c "
try:
    import matplotlib
    print(f'✅ matplotlib {matplotlib.__version__}')
except ImportError:
    print('❌ matplotlib not available')

try:
    import seaborn as sns
    print(f'✅ seaborn {sns.__version__}')
except ImportError:
    print('❌ seaborn not available')

try:
    import plotly
    print(f'✅ plotly {plotly.__version__}')
except ImportError:
    print('❌ plotly not available')
"
```

#### **Step 3.2: Test Lazy Import System**
```bash
# Test enhanced lazy import functions
poetry run python -c "
from giflab.lazy_imports import (
    is_matplotlib_available, is_seaborn_available, is_plotly_available,
    get_matplotlib, get_seaborn, get_plotly
)

print(f'Matplotlib available: {is_matplotlib_available()}')
print(f'Seaborn available: {is_seaborn_available()}')
print(f'Plotly available: {is_plotly_available()}')

if is_matplotlib_available():
    plt = get_matplotlib().pyplot
    print('✅ Matplotlib lazy loading working')
"
```

#### **Step 3.3: Validate Dependency Reporting**
```bash
# Check comprehensive dependency status
poetry run python -m giflab deps check --verbose
# Should show all visualization libraries in appropriate sections

# Test installation help
poetry run python -m giflab deps install-help matplotlib
poetry run python -m giflab deps install-help seaborn
```

**✅ Phase 3 Complete When:**
- Visualization libraries installed (optional)
- Lazy import system recognizes libraries
- Dependency reporting shows correct status

---

## Dependency Management Best Practices

### **Poetry Configuration**

#### **Recommended pyproject.toml Structure**
```toml
[tool.poetry.dependencies]
python = "^3.8"

# Core dependencies (required)
click = "^8.0.0"
numpy = "^1.20.0"
opencv-python = "^4.5.0"
Pillow = "^8.0.0"
rich = "^13.7.0"

# Processing dependencies  
pytest = "^6.0.0"

# Optional dependencies
psutil = {version = "^5.8.0", optional = true}
matplotlib = {version = "^3.0.0", optional = true}
seaborn = {version = "^0.11.0", optional = true}
plotly = {version = "^5.0.0", optional = true}
torch = {version = "^1.9.0", optional = true}
lpips = {version = "^0.1.0", optional = true}

[tool.poetry.extras]
monitoring = ["psutil"]
visualization = ["matplotlib", "seaborn", "plotly"]
ml = ["torch", "lpips"]
all = ["psutil", "matplotlib", "seaborn", "plotly", "torch", "lpips"]
```

#### **Installation Commands by Use Case**
```bash
# Minimal installation (core functionality only)
poetry install

# With memory monitoring
poetry install --extras monitoring

# With visualization support  
poetry install --extras visualization

# With ML capabilities
poetry install --extras ml

# Complete installation
poetry install --extras all
```

### **Environment Management**

#### **Clean Installation Procedure**
```bash
# Remove existing environment (if needed)
poetry env remove $(poetry env info --path)

# Create fresh environment
poetry install --extras all

# Verify clean installation
poetry run python -m giflab deps check --verbose
```

#### **Lock File Management**
```bash
# Update lock file after changes
poetry lock --no-update

# Update all dependencies to latest versions
poetry update

# Update specific dependency
poetry update rich matplotlib
```

---

## Troubleshooting

### **Common Issues**

#### **Rich Library Not Found**
```bash
# Issue: ImportError: No module named 'rich'
# Solution: Ensure rich is in pyproject.toml dependencies

# Check if rich is listed
grep "rich" pyproject.toml

# If missing, add it
poetry add rich@^13.7.0

# Reinstall
poetry install
```

#### **Memory Monitoring Unavailable**
```bash
# Issue: Memory monitoring features not working
# Solution: Install psutil

poetry add psutil

# Test installation
poetry run python -c "
import psutil
print('✅ psutil working')
print(f'Memory: {psutil.virtual_memory().percent:.1f}% used')
"
```

#### **CLI Formatting Issues**
```bash
# Issue: CLI output appears without rich formatting
# Possible causes: Terminal compatibility, rich version

# Check rich version
poetry run python -c "import rich; print(rich.__version__)"

# Test rich compatibility
poetry run python -c "
from rich.console import Console
console = Console()
console.print('Test rich output', style='bold green')
"

# Force rich update if needed
poetry add rich@^13.7.0 --force
```

#### **Visualization Libraries Conflicts**
```bash
# Issue: Matplotlib/seaborn conflicts or display issues
# Common with GUI backends

# Install with specific backend
poetry add matplotlib

# Configure non-GUI backend
poetry run python -c "
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
print('✅ matplotlib with Agg backend')
"
```

#### **Poetry Environment Issues**
```bash
# Issue: Poetry not finding virtual environment
# Solution: Recreate environment

# Check current environment
poetry env info

# List environments
poetry env list

# Remove problematic environment
poetry env remove $(poetry env info --path)

# Recreate
poetry install
```

### **Advanced Troubleshooting**

#### **Dependency Conflict Resolution**
```bash
# Check for conflicts
poetry check

# Show dependency tree
poetry show --tree

# Identify conflicting packages
poetry install --dry-run --verbose

# Force resolution (if safe)
poetry lock --no-update
poetry install
```

#### **Platform-Specific Issues**

**macOS Issues:**
```bash
# Issue: psutil compilation errors
# Solution: Install system dependencies
brew install python3-dev

# Issue: OpenCV issues
# Solution: Reinstall with specific options
poetry remove opencv-python
poetry add opencv-python-headless
```

**Linux Issues:**
```bash
# Issue: Missing system libraries for GUI
# Solution: Install system packages
sudo apt-get install python3-tk  # For matplotlib GUI
sudo apt-get install libgl1-mesa-glx  # For OpenCV

# Issue: Permission errors
# Solution: Fix Poetry permissions
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

**Windows Issues:**
```bash
# Issue: Long path errors
# Solution: Enable long paths or use short paths
poetry config virtualenvs.in-project true

# Issue: Visual C++ compiler errors
# Solution: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
```

---

## Validation Procedures

### **Comprehensive Dependency Check**
```bash
# Run full dependency validation
poetry run python -m giflab deps check --verbose --json > deps_validation.json

# Parse and analyze results
poetry run python -c "
import json
with open('deps_validation.json') as f:
    deps = json.load(f)

print('Dependency Validation Summary:')
print(f'Core dependencies: {len([d for d in deps[\"dependencies\"] if d[\"available\"]])} available')
print(f'Optional dependencies: {len([d for d in deps.get(\"optional_dependencies\", []) if d[\"available\"]])} available')

if deps.get('issues'):
    print(f'Issues found: {len(deps[\"issues\"])}')
    for issue in deps['issues']:
        print(f'  - {issue}')
else:
    print('✅ No dependency issues found')
"
```

### **Performance Impact Assessment**
```bash
# Measure dependency loading overhead
poetry run python -c "
import time

# Measure cold start time
start_time = time.time()
from giflab.cli import main
cold_start_time = time.time() - start_time

# Measure hot imports
start_time = time.time()
from giflab.metrics import calculate_comprehensive_metrics
from giflab.lazy_imports import get_matplotlib, is_pil_available
hot_import_time = time.time() - start_time

print(f'Cold start time: {cold_start_time*1000:.1f}ms')
print(f'Hot import time: {hot_import_time*1000:.1f}ms')

if cold_start_time > 2.0:  # More than 2 seconds
    print('⚠️ Slow cold start detected')
else:
    print('✅ Import performance acceptable')
"
```

### **System Integration Test**
```bash
# Test all systems work together
poetry run python -c "
print('Testing integrated systems...')

# Test CLI with dependencies
import subprocess
result = subprocess.run(['poetry', 'run', 'python', '-m', 'giflab', 'deps', 'status'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print('✅ CLI integration working')
else:
    print(f'❌ CLI integration failed: {result.stderr}')

# Test memory monitoring if available
try:
    from giflab.monitoring import SystemMemoryMonitor
    monitor = SystemMemoryMonitor()
    stats = monitor.get_memory_stats()
    print(f'✅ Memory monitoring: {stats.available_gb:.1f}GB available')
except Exception as e:
    print(f'⚠️ Memory monitoring issue: {e}')

# Test lazy imports
from giflab.lazy_imports import is_matplotlib_available, is_seaborn_available
print(f'✅ Visualization: matplotlib={is_matplotlib_available()}, seaborn={is_seaborn_available()}')

print('Integration test complete.')
"
```

---

## Rollback Procedures

### **Dependency Rollback**

#### **Remove Optional Dependencies**
```bash
# Remove optional visualization libraries
poetry remove matplotlib seaborn plotly --dry-run  # Preview changes
poetry remove matplotlib seaborn plotly

# Remove memory monitoring
poetry remove psutil

# Update lock file
poetry lock --no-update
```

#### **Restore Minimal Environment**
```bash
# Reset to minimal dependency set
poetry env remove $(poetry env info --path)
poetry install  # Only core dependencies

# Verify minimal installation
poetry run python -m giflab --help
poetry run python -c "from giflab.metrics import calculate_comprehensive_metrics"
```

#### **Emergency Dependency Recovery**
```bash
# If Poetry environment corrupted
poetry env remove --all
poetry cache clear --all pypi
poetry install --no-cache

# Verify recovery
poetry run python -m giflab deps check
```

### **Configuration Rollback**

#### **Disable Enhanced Features**
```python
# In src/giflab/config.py - disable memory monitoring
MONITORING = {
    "memory_pressure": {
        "enabled": False,  # Disable memory monitoring
    }
}

# Keep caching disabled (should already be False)
ENABLE_EXPERIMENTAL_CACHING = False
```

---

## Future Dependency Management

### **Adding New Dependencies**

#### **Evaluation Criteria**
Before adding new dependencies, evaluate:
1. **Necessity**: Is this functionality critical?
2. **Alternatives**: Can we achieve this with existing dependencies?
3. **Maintenance**: Is the package actively maintained?
4. **Size**: What's the installation footprint?
5. **Compatibility**: Does it conflict with existing dependencies?

#### **Addition Process**
```bash
# 1. Research dependency
poetry search <package-name>

# 2. Add as optional first
poetry add <package-name> --optional

# 3. Test integration
poetry run python -c "import <package>; print('✅ Working')"

# 4. Add to appropriate extras group in pyproject.toml

# 5. Update lazy imports if needed
# Add to src/giflab/lazy_imports.py

# 6. Update dependency checking
# Add to src/giflab/cli/deps_cmd.py

# 7. Test comprehensive dependency check
poetry run python -m giflab deps check --verbose
```

### **Dependency Monitoring**

#### **Regular Maintenance Tasks**
```bash
# Monthly: Check for updates
poetry show --outdated

# Quarterly: Update non-breaking versions  
poetry update --dry-run
poetry update

# Annually: Major version upgrades (requires testing)
poetry add <package>@^<new-major-version> --dry-run
```

#### **Security Monitoring**
```bash
# Check for security issues (requires poetry-plugin-audit)
poetry audit

# Update security-critical packages immediately
poetry update <security-critical-package>
```

---

## Summary

### **Required Actions**
1. **Install Rich**: Core dependency for CLI formatting
2. **Run poetry install**: Ensure all dependencies current
3. **Test CLI functionality**: Verify enhanced formatting works

### **Recommended Actions**
1. **Install psutil**: Enable memory monitoring capabilities
2. **Install visualization libraries**: Enhanced analysis capabilities
3. **Set up dependency monitoring**: Regular maintenance scheduling

### **Optional Actions**
1. **Configure Poetry extras**: Organized dependency groups
2. **Set up security monitoring**: Automated vulnerability checking
3. **Performance monitoring**: Track dependency loading overhead

### **Migration Effort**
- **Core Dependencies**: 10-15 minutes (mostly automated)
- **Optional Dependencies**: 15-30 minutes (user choice dependent)
- **Testing & Validation**: 15-30 minutes
- **Total Time**: 30-75 minutes depending on optional features selected

### **Success Metrics**
- All core dependencies installed and working
- CLI commands display with rich formatting
- Dependency checking commands functional (`giflab deps`)
- No performance regression in core operations
- Optional features work when dependencies available

---

*This dependency migration guide provides comprehensive procedures for managing dependencies during the Phases 1-4 architectural improvements. The guide emphasizes safety, testing, and graceful degradation for optional dependencies.*