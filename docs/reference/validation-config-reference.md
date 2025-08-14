# 📋 Validation Configuration Reference

Complete reference guide for all validation system configuration options, parameters, and usage patterns.

---

## 🏗️ ValidationConfig Class

### Class Definition

```python
from dataclasses import dataclass

@dataclass
class ValidationConfig:
    """Configuration for wrapper output validation system."""
    
    # System Control
    ENABLE_WRAPPER_VALIDATION: bool = True
    LOG_VALIDATION_FAILURES: bool = False
    FAIL_ON_VALIDATION_ERROR: bool = False
    
    # Frame Validation
    FRAME_RATIO_TOLERANCE: float = 0.05
    MIN_FRAMES_REQUIRED: int = 1
    
    # Color Validation
    COLOR_COUNT_TOLERANCE: int = 2
    MIN_COLOR_REDUCTION_PERCENT: float = 0.05
    
    # Timing Validation
    FPS_TOLERANCE: float = 0.1
    MIN_FPS: float = 0.1
    MAX_FPS: float = 60.0
    
    # File Integrity
    MIN_FILE_SIZE_BYTES: int = 100
    MAX_FILE_SIZE_MB: float = 50.0
```

---

## ⚙️ Configuration Parameters

### System Control Parameters

#### `ENABLE_WRAPPER_VALIDATION: bool`
**Default**: `True`  
**Purpose**: Master switch for the entire validation system  
**Impact**: When `False`, validation is completely bypassed

```python
# Disable validation entirely
config = ValidationConfig(ENABLE_WRAPPER_VALIDATION=False)

# Enable validation (default)
config = ValidationConfig(ENABLE_WRAPPER_VALIDATION=True)
```

**Use Cases**:
- ✅ **Production emergency**: Disable validation if causing issues
- ✅ **Performance testing**: Measure pure compression performance
- ❌ **Normal operation**: Keep enabled for data integrity

#### `LOG_VALIDATION_FAILURES: bool`
**Default**: `False`  
**Purpose**: Controls whether validation failures are logged  
**Impact**: When `True`, failed validations are logged as warnings

```python
# Enable validation failure logging
config = ValidationConfig(LOG_VALIDATION_FAILURES=True)

# Example log output:
# WARNING: Validation failures for color_reduction on output.gif: 
#   ['Color count 35 exceeds expected 32 + tolerance 2']
```

**Use Cases**:
- ✅ **Development/debugging**: Track validation issues
- ✅ **Quality monitoring**: Monitor validation failure rates
- ❌ **High-throughput production**: Avoid I/O overhead

#### `FAIL_ON_VALIDATION_ERROR: bool`
**Default**: `False` ⚠️  
**Purpose**: Whether to raise exceptions on validation failures  
**Impact**: When `True`, validation failures raise `RuntimeError`

```python
# Never recommended - breaks pipeline execution
config = ValidationConfig(FAIL_ON_VALIDATION_ERROR=True)
```

**⚠️ Important**: Almost always keep this `False`. Validation is designed to be informational, not blocking.

---

### Frame Validation Parameters

#### `FRAME_RATIO_TOLERANCE: float`
**Default**: `0.05` (5%)  
**Range**: `0.0` to `1.0`  
**Purpose**: Acceptable deviation from expected frame reduction ratio

```python
# Strict frame validation (1% tolerance)
strict_config = ValidationConfig(FRAME_RATIO_TOLERANCE=0.01)

# Relaxed frame validation (10% tolerance)
relaxed_config = ValidationConfig(FRAME_RATIO_TOLERANCE=0.10)

# Example: With 0.05 tolerance
# Expected: 50% reduction (10 frames → 5 frames)
# Acceptable: 47.5% to 52.5% reduction (4-6 frames)
```

**Tuning Guidelines**:
- **0.01-0.02**: Very strict, for critical applications
- **0.05**: Default, good balance
- **0.10-0.20**: Relaxed, for performance or when frame count precision isn't critical

#### `MIN_FRAMES_REQUIRED: int`
**Default**: `1`  
**Range**: `1` to `∞`  
**Purpose**: Minimum number of frames required in output

```python
# Require at least 3 frames (prevent over-reduction)
config = ValidationConfig(MIN_FRAMES_REQUIRED=3)

# Allow single-frame outputs (default)
config = ValidationConfig(MIN_FRAMES_REQUIRED=1)
```

**Use Cases**:
- **Animation preservation**: Ensure GIFs remain animated
- **Quality control**: Prevent excessive frame reduction
- **Format requirements**: Meet specific output constraints

---

### Color Validation Parameters

#### `COLOR_COUNT_TOLERANCE: int`
**Default**: `2` colors  
**Range**: `0` to `∞`  
**Purpose**: Acceptable excess colors beyond target count

```python
# Exact color matching
strict_config = ValidationConfig(COLOR_COUNT_TOLERANCE=0)

# Allow up to 5 extra colors
lenient_config = ValidationConfig(COLOR_COUNT_TOLERANCE=5)

# Example: With tolerance=2, target=32
# Acceptable: ≤34 colors in output
# Failure: ≥35 colors in output
```

**Tuning Guidelines**:
- **0**: Exact matching, very strict
- **2**: Default, accounts for encoding variations
- **5-10**: Relaxed, for complex images or performance

#### `MIN_COLOR_REDUCTION_PERCENT: float`
**Default**: `0.05` (5%)  
**Range**: `0.0` to `1.0`  
**Purpose**: Minimum required color reduction percentage

```python
# Require at least 10% color reduction
config = ValidationConfig(MIN_COLOR_REDUCTION_PERCENT=0.10)

# No reduction requirement (allow color increases)
config = ValidationConfig(MIN_COLOR_REDUCTION_PERCENT=0.0)

# Example: With 5% requirement
# Input: 256 colors, Target: 64 colors
# Expected reduction: (256-64)/256 = 75%
# Minimum acceptable: 5% reduction = 243 colors maximum
```

---

### Timing Validation Parameters

#### `FPS_TOLERANCE: float`
**Default**: `0.1` (10%)  
**Range**: `0.0` to `1.0`  
**Purpose**: Acceptable FPS variation from input to output

```python
# Strict FPS preservation (5% tolerance)
config = ValidationConfig(FPS_TOLERANCE=0.05)

# Relaxed FPS tolerance (20% tolerance) 
config = ValidationConfig(FPS_TOLERANCE=0.2)

# Example: Input at 10 FPS, tolerance=0.1
# Acceptable range: 9.0 to 11.0 FPS
```

#### `MIN_FPS: float` & `MAX_FPS: float`
**Defaults**: `0.1` and `60.0`  
**Purpose**: Valid FPS range bounds

```python
# Narrow FPS range for specific use case
config = ValidationConfig(
    MIN_FPS=5.0,    # At least 5 FPS
    MAX_FPS=30.0    # At most 30 FPS
)

# Wide range for general use
config = ValidationConfig(
    MIN_FPS=0.1,    # Very slow animations OK
    MAX_FPS=120.0   # High frame rate OK
)
```

---

### File Integrity Parameters

#### `MIN_FILE_SIZE_BYTES: int`
**Default**: `100` bytes  
**Purpose**: Minimum acceptable output file size

```python
# Require larger minimum files
config = ValidationConfig(MIN_FILE_SIZE_BYTES=1000)  # 1KB minimum

# Very permissive (almost any non-empty file)
config = ValidationConfig(MIN_FILE_SIZE_BYTES=10)    # 10 bytes minimum
```

#### `MAX_FILE_SIZE_MB: float`  
**Default**: `50.0` MB  
**Purpose**: Maximum acceptable output file size

```python
# Limit large file processing
config = ValidationConfig(MAX_FILE_SIZE_MB=5.0)      # 5MB maximum

# Allow very large files
config = ValidationConfig(MAX_FILE_SIZE_MB=100.0)    # 100MB maximum
```

**Performance Note**: Large files significantly increase validation time.

---

## 📋 Pre-configured Settings

### `DEFAULT_VALIDATION_CONFIG`
General-purpose configuration suitable for most applications.

```python
from giflab.config import DEFAULT_VALIDATION_CONFIG

# Access default configuration
config = DEFAULT_VALIDATION_CONFIG

# All default values:
# ENABLE_WRAPPER_VALIDATION=True
# FRAME_RATIO_TOLERANCE=0.05
# COLOR_COUNT_TOLERANCE=2
# etc.
```

---

## 🎯 Configuration Recipes

### Recipe 1: Development Configuration
**Use Case**: Development, testing, debugging  
**Priority**: Maximum issue detection

```python
DEV_CONFIG = ValidationConfig(
    ENABLE_WRAPPER_VALIDATION=True,
    LOG_VALIDATION_FAILURES=True,           # Log all issues
    
    # Strict tolerances to catch problems early
    FRAME_RATIO_TOLERANCE=0.01,             # 1% tolerance
    COLOR_COUNT_TOLERANCE=0,                # Exact color matching
    FPS_TOLERANCE=0.05,                     # 5% FPS tolerance
    MIN_COLOR_REDUCTION_PERCENT=0.1,        # Require 10% color reduction
    
    # Process all file sizes
    MIN_FILE_SIZE_BYTES=1,
    MAX_FILE_SIZE_MB=100.0,
)
```

### Recipe 2: Production Configuration
**Use Case**: High-throughput production workloads  
**Priority**: Performance with essential validation

```python
PRODUCTION_CONFIG = ValidationConfig(
    ENABLE_WRAPPER_VALIDATION=True,
    LOG_VALIDATION_FAILURES=False,          # No I/O overhead
    
    # Relaxed tolerances for performance
    FRAME_RATIO_TOLERANCE=0.15,             # 15% tolerance
    COLOR_COUNT_TOLERANCE=10,               # 10 color tolerance
    FPS_TOLERANCE=0.2,                      # 20% FPS tolerance
    MIN_COLOR_REDUCTION_PERCENT=0.01,       # Minimal reduction requirement
    
    # Limit file sizes for consistent performance
    MIN_FILE_SIZE_BYTES=500,
    MAX_FILE_SIZE_MB=10.0,
)
```

### Recipe 3: Quality Assurance Configuration
**Use Case**: QA testing, compliance verification  
**Priority**: Comprehensive validation coverage

```python
QA_CONFIG = ValidationConfig(
    ENABLE_WRAPPER_VALIDATION=True,
    LOG_VALIDATION_FAILURES=True,           # Log everything
    FAIL_ON_VALIDATION_ERROR=False,         # Never break pipeline
    
    # Moderate tolerances - strict enough to catch real issues
    FRAME_RATIO_TOLERANCE=0.03,             # 3% tolerance
    COLOR_COUNT_TOLERANCE=1,                # 1 color tolerance
    FPS_TOLERANCE=0.08,                     # 8% FPS tolerance
    MIN_COLOR_REDUCTION_PERCENT=0.05,       # 5% reduction requirement
    
    # Wide file size range for comprehensive testing
    MIN_FILE_SIZE_BYTES=50,
    MAX_FILE_SIZE_MB=25.0,
)
```

### Recipe 4: Interactive Application Configuration
**Use Case**: Real-time applications, user interfaces  
**Priority**: Low latency with basic validation

```python
INTERACTIVE_CONFIG = ValidationConfig(
    ENABLE_WRAPPER_VALIDATION=True,
    LOG_VALIDATION_FAILURES=False,          # No I/O delays
    
    # Fast validations with reasonable tolerances
    FRAME_RATIO_TOLERANCE=0.2,              # 20% tolerance
    COLOR_COUNT_TOLERANCE=5,                # 5 color tolerance
    FPS_TOLERANCE=0.3,                      # 30% FPS tolerance
    MIN_COLOR_REDUCTION_PERCENT=0.0,        # No reduction requirement
    
    # Limit processing to small files for responsiveness
    MIN_FILE_SIZE_BYTES=100,
    MAX_FILE_SIZE_MB=2.0,                   # 2MB maximum
)
```

### Recipe 5: Batch Processing Configuration
**Use Case**: Large-scale batch processing  
**Priority**: Throughput with selective validation

```python
BATCH_CONFIG = ValidationConfig(
    ENABLE_WRAPPER_VALIDATION=True,
    LOG_VALIDATION_FAILURES=False,          # Minimize I/O
    
    # Very relaxed tolerances for speed
    FRAME_RATIO_TOLERANCE=0.25,             # 25% tolerance
    COLOR_COUNT_TOLERANCE=20,               # 20 color tolerance
    FPS_TOLERANCE=0.4,                      # 40% FPS tolerance
    MIN_COLOR_REDUCTION_PERCENT=0.0,        # No requirements
    
    # Process reasonable file sizes efficiently
    MIN_FILE_SIZE_BYTES=200,
    MAX_FILE_SIZE_MB=5.0,                   # 5MB maximum
)
```

---

## 🔧 Configuration Usage Patterns

### Pattern 1: Environment-based Configuration

```python
import os
from giflab.config import ValidationConfig, DEFAULT_VALIDATION_CONFIG

def get_validation_config():
    """Get validation config based on environment."""
    env = os.getenv("GIFLAB_ENV", "development")
    
    if env == "production":
        return PRODUCTION_CONFIG
    elif env == "qa":
        return QA_CONFIG
    elif env == "development":
        return DEV_CONFIG
    else:
        return DEFAULT_VALIDATION_CONFIG

# Usage
config = get_validation_config()
```

### Pattern 2: Wrapper-specific Configuration

```python
class HighPrecisionWrapper:
    def __init__(self):
        # Custom config for this specific wrapper
        self.validation_config = ValidationConfig(
            FRAME_RATIO_TOLERANCE=0.001,        # Very precise
            COLOR_COUNT_TOLERANCE=0,            # Exact
            LOG_VALIDATION_FAILURES=True       # Debug this wrapper
        )
    
    def apply(self, input_path, output_path, params):
        result = self._compress(input_path, output_path, params)
        return add_validation_to_result(
            input_path, output_path, params, result,
            wrapper_type="frame_reduction",
            config=self.validation_config      # Use custom config
        )
```

### Pattern 3: Dynamic Configuration

```python
class AdaptiveWrapper:
    def get_dynamic_config(self, file_size: int, operation_type: str):
        """Adjust config based on file characteristics."""
        
        if file_size > 5_000_000:  # Files > 5MB
            # Use performance config for large files
            return ValidationConfig(
                FRAME_RATIO_TOLERANCE=0.2,
                COLOR_COUNT_TOLERANCE=10,
                LOG_VALIDATION_FAILURES=False
            )
        elif operation_type == "color_reduction":
            # Stricter color validation
            return ValidationConfig(
                COLOR_COUNT_TOLERANCE=0,
                MIN_COLOR_REDUCTION_PERCENT=0.1
            )
        else:
            # Default for small files
            return DEFAULT_VALIDATION_CONFIG
    
    def apply(self, input_path, output_path, params):
        file_size = input_path.stat().st_size
        config = self.get_dynamic_config(file_size, self._get_operation_type())
        
        result = self._compress(input_path, output_path, params)
        return add_validation_to_result(
            input_path, output_path, params, result,
            config=config
        )
```

---

## ✅ Configuration Validation

### Validating Configuration Values

```python
def validate_config(config: ValidationConfig) -> list[str]:
    """Validate configuration for common issues."""
    warnings = []
    
    # Check tolerance ranges
    if config.FRAME_RATIO_TOLERANCE > 0.5:
        warnings.append("FRAME_RATIO_TOLERANCE > 50% may be too permissive")
    
    if config.COLOR_COUNT_TOLERANCE > 50:
        warnings.append("COLOR_COUNT_TOLERANCE > 50 may be too permissive")
    
    if config.FPS_TOLERANCE > 0.5:
        warnings.append("FPS_TOLERANCE > 50% may be too permissive")
    
    # Check file size limits
    if config.MAX_FILE_SIZE_MB > 100:
        warnings.append("MAX_FILE_SIZE_MB > 100MB may impact performance")
    
    if config.MIN_FILE_SIZE_BYTES > 10000:
        warnings.append("MIN_FILE_SIZE_BYTES > 10KB may skip valid small files")
    
    # Check dangerous settings
    if config.FAIL_ON_VALIDATION_ERROR:
        warnings.append("FAIL_ON_VALIDATION_ERROR=True may break pipelines")
    
    return warnings

# Example usage
config = ValidationConfig(FRAME_RATIO_TOLERANCE=0.8)  # Very permissive
warnings = validate_config(config)
for warning in warnings:
    print(f"⚠️  {warning}")
```

---

## 🚨 Common Configuration Mistakes

### Mistake 1: Over-strict Tolerances
```python
# ❌ Too strict - will cause many false positives
bad_config = ValidationConfig(
    FRAME_RATIO_TOLERANCE=0.001,    # 0.1% - almost impossible to meet
    COLOR_COUNT_TOLERANCE=0,        # Exact - encoding variations will fail
    FPS_TOLERANCE=0.01             # 1% - very strict for floating-point FPS
)

# ✅ Better - realistic tolerances
good_config = ValidationConfig(
    FRAME_RATIO_TOLERANCE=0.05,     # 5% - accounts for rounding
    COLOR_COUNT_TOLERANCE=2,        # Small allowance for encoding
    FPS_TOLERANCE=0.1               # 10% - reasonable for FPS variations
)
```

### Mistake 2: Performance-killing Settings
```python
# ❌ Performance issues
bad_config = ValidationConfig(
    LOG_VALIDATION_FAILURES=True,   # I/O overhead in production
    MAX_FILE_SIZE_MB=500.0,         # Will process huge files
    MIN_FILE_SIZE_BYTES=1           # Will process tiny files
)

# ✅ Performance-aware
good_config = ValidationConfig(
    LOG_VALIDATION_FAILURES=False,  # No I/O overhead
    MAX_FILE_SIZE_MB=10.0,          # Reasonable limit
    MIN_FILE_SIZE_BYTES=100         # Skip tiny files
)
```

### Mistake 3: Dangerous Error Handling
```python
# ❌ NEVER do this
dangerous_config = ValidationConfig(
    FAIL_ON_VALIDATION_ERROR=True   # Will break pipelines!
)

# ✅ Always use safe error handling  
safe_config = ValidationConfig(
    FAIL_ON_VALIDATION_ERROR=False, # Validation is informational
    LOG_VALIDATION_FAILURES=True   # Log issues for review
)
```

---

## 📊 Configuration Impact Analysis

| Parameter | Low Value Impact | High Value Impact | Recommended Range |
|-----------|------------------|-------------------|-------------------|
| `FRAME_RATIO_TOLERANCE` | More false positives | Less precision | 0.01 - 0.15 |
| `COLOR_COUNT_TOLERANCE` | Strict color matching | Permissive counting | 0 - 10 |
| `FPS_TOLERANCE` | Strict timing | Flexible timing | 0.05 - 0.3 |
| `MIN_FRAMES_REQUIRED` | More restrictive | Less restrictive | 1 - 5 |
| `MAX_FILE_SIZE_MB` | Skip large files | Process all files | 1 - 50 |
| `LOG_VALIDATION_FAILURES` | No logging overhead | Full debugging info | Context-dependent |

---

## 📚 Related Documentation

- [Wrapper Integration Guide](../guides/wrapper-validation-integration.md)
- [Performance Optimization Guide](../technical/validation-performance-guide.md)
- [Troubleshooting Guide](../guides/validation-troubleshooting.md)
- [Quality Metrics Integration](../technical/validation-quality-integration.md)

---

## 🔗 Configuration Examples Repository

All configuration examples in this document are available in:
- `tests/test_config_examples.py` - Unit tests for configuration recipes
- `examples/validation_configs.py` - Ready-to-use configuration examples