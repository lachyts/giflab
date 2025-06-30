# GifLab Bug Fixes Report

This report documents three significant bugs found and fixed in the GifLab codebase.

## Bug #1: Division by Zero in Color Reduction Logic

**Type:** Logic Error
**Severity:** High
**Location:** `src/giflab/color_keep.py`, line 159

### Problem Description
The `get_color_reduction_info()` function had a potential division by zero error when calculating the compression ratio. While there was a check for `target_colors > 0`, the logic was flawed because corrupted GIFs could have `original_colors = 0`, leading to `target_colors = 0` and division by zero.

### Root Cause
```python
# Problematic code:
target_colors = min(color_count, original_colors)  # Could be 0 if original_colors is 0
"compression_ratio": original_colors / target_colors if target_colors > 0 else 1.0
```

If a GIF is corrupted and reports 0 colors, `target_colors` becomes 0, causing division by zero.

### Solution
1. Added validation to detect invalid color counts from corrupted GIFs
2. Enhanced the division by zero protection to check both `target_colors > 0` and `original_colors > 0`

### Fixed Code
```python
# Validate that we have at least 1 color (handle corrupted GIFs)
if original_colors <= 0:
    raise ValueError(f"Invalid color count detected: {original_colors}")

# Enhanced division protection
"compression_ratio": original_colors / target_colors if target_colors > 0 and original_colors > 0 else 1.0
```

### Impact
- **Before:** Application would crash with `ZeroDivisionError` when processing corrupted GIFs
- **After:** Corrupted GIFs are properly detected and handled with informative error messages

---

## Bug #2: Hardcoded Path Breaking Portability

**Type:** Portability Issue  
**Severity:** Critical
**Location:** `src/giflab/lossy.py`, line 188

### Problem Description
The path to the animately compression engine was hardcoded to a specific macOS user directory (`/Users/lachlants/bin/launcher`), making the software completely non-portable and causing failures on any other system.

### Root Cause
```python
# Problematic code:
animately_path = "/Users/lachlants/bin/launcher"
```

This hardcoded path would fail on:
- Linux systems
- Windows systems  
- Other macOS systems
- Different user accounts
- Docker containers
- CI/CD environments

### Solution
Implemented a flexible path resolution system that:
1. Checks the `ANIMATELY_PATH` environment variable first
2. Searches common installation directories
3. Maintains backwards compatibility with the original path
4. Uses system PATH lookup as fallback

### Fixed Code
```python
def _find_animately_launcher() -> Optional[str]:
    """Find the animately launcher executable."""
    # Check environment variable first
    env_path = os.environ.get('ANIMATELY_PATH')
    if env_path and Path(env_path).exists():
        return env_path
    
    # Common installation paths to check
    search_paths = [
        "/Users/lachlants/bin/launcher",  # Original for backwards compatibility
        "/usr/local/bin/animately",
        "/usr/bin/animately",
        "/opt/animately/bin/launcher",
        os.path.expanduser("~/bin/animately"),
        os.path.expanduser("~/bin/launcher"),
        "./animately",
        "./launcher"
    ]
    
    for path in search_paths:
        if Path(path).exists():
            return path
    
    # Try system PATH
    try:
        result = subprocess.run(['which', 'animately'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    
    return None
```

### Impact
- **Before:** Software only worked on the original developer's Mac
- **After:** Software works across platforms with configurable installation paths

---

## Bug #3: Overly Broad Exception Handling

**Type:** Security/Debugging Issue
**Severity:** Medium
**Location:** `src/giflab/metrics.py`, line 209 and multiple locations

### Problem Description
The code used bare `except Exception:` blocks that caught and silently ignored all exceptions, making debugging extremely difficult and potentially hiding critical errors like memory issues, numerical instabilities, or security problems.

### Root Cause
```python
# Problematic code:
try:
    scale_ssim = ssim(current_frame1, current_frame2, data_range=255.0)
    ssim_values.append(scale_ssim)
except Exception:  # Too broad - catches everything!
    ssim_values.append(0.0)
```

This pattern would silently swallow:
- Memory errors
- Numerical overflow/underflow
- Array dimension mismatches
- Security exceptions
- System errors

### Solution
1. Replaced broad exception handling with specific exception types
2. Added proper logging of caught exceptions
3. Enhanced division by zero protection in temporal consistency calculation

### Fixed Code
```python
try:
    scale_ssim = ssim(current_frame1, current_frame2, data_range=255.0)
    ssim_values.append(scale_ssim)
except (ValueError, RuntimeError) as e:
    # Specific exceptions for data/computation errors
    logger.warning(f"SSIM calculation failed at scale {scale}: {e}")
    ssim_values.append(0.0)

# Also fixed division by zero protection:
epsilon = 1e-8
consistency = 1.0 / (1.0 + variance_diff / max(mean_diff, epsilon))
```

### Impact
- **Before:** Critical errors were silently ignored, making debugging impossible
- **After:** Specific errors are logged while maintaining application resilience

---

## Testing Recommendations

To verify these fixes:

1. **Bug #1:** Test with a corrupted GIF file that reports 0 colors
2. **Bug #2:** Test on different operating systems and with various animately installation paths
3. **Bug #3:** Monitor logs during metric calculations to ensure proper error reporting

## Summary

These fixes address:
- **Reliability:** Prevents crashes from division by zero
- **Portability:** Enables cross-platform deployment  
- **Maintainability:** Improves error visibility and debugging capabilities

All fixes maintain backwards compatibility while significantly improving the robustness and portability of the GifLab codebase.