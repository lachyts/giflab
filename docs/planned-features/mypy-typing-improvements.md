# MyPy Typing Improvements - Phase 1+ Planning

## Overview

This document outlines MyPy type checking improvements for the GifLab codebase. The goal is to systematically reduce type errors while maintaining 100% test compatibility and zero functional regressions.

## Current Status Assessment

Run the following command to get current MyPy error count:
```bash
poetry run mypy src/ --ignore-missing-imports | grep "error" | wc -l
```

## Phase 1: Strategic Next Steps

### 🎯 Priority 1: Core Function Signatures 

#### Step 1.1: Add Basic Type Annotations
**Target**: Add missing type annotations to function parameters and return values

**Approach**:
1. Start with core modules: `metrics.py`, `pipeline.py`, `io.py`
2. Add type annotations to public functions first
3. Focus on functions with clear parameter/return types

**Example fixes needed**:
```python
# Before:
def complex_function(data, callback=None, **kwargs):  # ❌
    return process_data(data)

# After:  
def complex_function(data: dict[str, Any], callback: Callable[[str], None] | None = None, **kwargs: Any) -> dict[str, Any]:  # ✅
    return process_data(data)
```

**Implementation**:
1. Run `poetry run mypy src/giflab/metrics.py` to identify errors
2. Add type annotations systematically 
3. Test with `poetry run pytest tests/ -x` after each file
4. Use `from typing import Any, Callable, Dict, List, Optional, Union` as needed

**Success Criteria**: 
- All public functions in core modules have type annotations
- Zero test regressions
- MyPy errors reduced by 30-50

#### Step 1.2: Variable Type Annotations  
**Target**: Add explicit type annotations to class attributes and module-level variables

**Approach**:
1. Focus on variables that cause the most MyPy errors
2. Add type annotations to class attributes in `__init__` methods
3. Annotate module-level constants and configuration variables

**Example fixes needed**:
```python
# Before:
class Pipeline:
    def __init__(self):
        self.metrics = {}  # ❌ Implicit Any type
        self.results = []   # ❌ Implicit Any type

# After:
class Pipeline:
    def __init__(self):
        self.metrics: dict[str, float] = {}  # ✅ Explicit type
        self.results: list[dict[str, Any]] = []   # ✅ Explicit type
```

**Implementation**:
1. Run `poetry run mypy src/giflab/pipeline.py` to identify variable type issues
2. Add type annotations to class attributes and instance variables  
3. Focus on containers (lists, dicts) that store business data
4. Use `from typing import Any, Dict, List` imports

**Success Criteria**:
- All class attributes have explicit types
- Module-level variables are properly typed  
- MyPy errors reduced by additional 20-30

## Phase 2: Advanced Type Safety

### 🔧 Priority 2: Union Types and Optional Handling

#### Step 2.1: Fix None/Optional Type Issues
**Target**: Resolve type errors related to None values and optional parameters

**Common Issues**:
```python
# Before:
def process_data(data):
    if data is None:
        return None
    return data.process()  # ❌ MyPy: data could be None

# After:
def process_data(data: Optional[DataType]) -> Optional[ProcessedType]:
    if data is None:
        return None
    return data.process()  # ✅ MyPy understands None check
```

**Implementation**:
1. Identify functions that can return None or accept None parameters
2. Add `Optional[Type]` or `Type | None` annotations  
3. Add proper None checks where needed
4. Use `from typing import Optional` import

**Success Criteria**:
- All functions with None handling properly typed
- No "possibly None" MyPy errors
- Maintain existing None-safety behavior

#### Step 2.2: Fix PIL Image Type Issues  
**Target**: Resolve image type inconsistencies in multiprocessing code

**Common Issues**:
```python
# Before:
images = []  # ❌ Implicit Any type
images.append(Image.open(path))  # ❌ Type unclear

# After:
images: list[Image.Image] = []  # ✅ Explicit PIL Image type
images.append(Image.open(path))  # ✅ Compatible types
```

**Implementation**:
1. Add `from PIL import Image` imports consistently
2. Use `Image.Image` type for PIL image objects
3. Add proper type annotations to image processing functions
4. Handle ImageFile vs Image.Image distinctions

**Files to focus on**: `multiprocessing_support.py`, `synthetic_gifs.py`, `io.py`

**Success Criteria**:
- All image variables properly typed
- No PIL-related MyPy errors
- Multiprocessing image handling works correctly

## Phase 3: External Library Integration

### 🏗️ Priority 3: Third-Party Library Types

#### Step 3.1: Install Type Stubs  
**Target**: Install official type stubs for major libraries

**Installation**:
```bash
# Install official stubs for common libraries
poetry add --group dev types-PIL types-requests pandas-stubs

# Check what stubs are available
poetry search types- | head -20
```

**Implementation**:
1. Start with libraries that have official stubs
2. Install stubs one at a time and test MyPy output
3. Focus on libraries causing the most errors

**Success Criteria**:
- Major libraries have proper type support
- MyPy can resolve library function signatures
- No regressions in functionality

#### Step 3.2: Handle OpenCV/GPU Type Issues
**Target**: Address cv2 module attribute errors

**Common OpenCV Issues**:
```python
# Problem: MyPy doesn't recognize OpenCV attributes
cv2.cuda.getCudaEnabledDeviceCount()  # ❌ Module has no attribute

# Solutions:
# Option 1: Strategic type ignore
device_count = cv2.cuda.getCudaEnabledDeviceCount()  # type: ignore[attr-defined]

# Option 2: Try/except with typing  
try:
    device_count = cv2.cuda.getCudaEnabledDeviceCount()
except AttributeError:
    device_count = 0  # Fallback for systems without CUDA
```

**Implementation**:
1. Focus on GPU-related cv2 functions first
2. Add `# type: ignore[attr-defined]` for known OpenCV functions
3. Consider conditional imports where appropriate
4. Document why ignores are necessary

**Files**: `experimental/runner.py`, `external_engines/*.py`

## Implementation Strategy

### Week 1: Phase 1 Foundation
**Goal**: Get basic type annotations in place
1. **Day 1-2**: Run MyPy baseline, implement Step 1.1 (function signatures)
2. **Day 3-4**: Implement Step 1.2 (variable annotations) 
3. **Day 5**: Test all changes, ensure zero test regressions

**Expected Result**: 40-60% of current MyPy errors resolved

### Week 2: Phase 2 Refinement  
**Goal**: Handle complex type scenarios
1. **Day 1-2**: Implement Step 2.1 (None/Optional handling)
2. **Day 3-4**: Implement Step 2.2 (PIL Image types)
3. **Day 5**: Integration testing and refinement

**Expected Result**: 70-80% of current MyPy errors resolved

### Week 3: Phase 3 Polish
**Goal**: External library integration
1. **Day 1-2**: Install type stubs (Step 3.1)
2. **Day 3-4**: Handle OpenCV issues (Step 3.2)
3. **Day 5**: Final validation and documentation

**Expected Result**: 85-95% of current MyPy errors resolved

## Success Metrics

### Quality Gates (Apply at each step)
- ✅ **Zero test regressions**: `poetry run pytest tests/ -x` must pass
- ✅ **Zero functional changes**: Only type annotations, no behavior changes
- ✅ **Incremental progress**: Each step reduces MyPy errors
- ✅ **Performance maintained**: No runtime type checking overhead

## Getting Started

### Prerequisites
```bash
# Ensure MyPy is available
poetry add --group dev mypy

# Get baseline error count
poetry run mypy src/ --ignore-missing-imports 2>&1 | grep "error" | wc -l
```

### Quick Start Commands
```bash
# Check specific module
poetry run mypy src/giflab/metrics.py --ignore-missing-imports

# Run tests to ensure no regressions  
poetry run pytest tests/ -x

# Check overall progress
poetry run mypy src/ --ignore-missing-imports | tail -5
```

### Development Workflow
1. **Before making changes**: Run tests to ensure green baseline
2. **While adding types**: Check individual files with MyPy
3. **After each change**: Re-run tests to catch any issues
4. **At end of session**: Check overall MyPy error reduction

## Tools and Resources

### Essential Commands
```bash
# Core MyPy workflow
poetry run mypy src/giflab/metrics.py          # Check single file
poetry run mypy src/ --ignore-missing-imports  # Check everything  
poetry run pytest tests/ -x                    # Verify no regressions

# Advanced analysis
poetry run mypy src/ --show-error-codes        # Show error types
poetry run mypy src/ --strict                  # Strict mode checking
```

### Useful References
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Python Typing Module](https://docs.python.org/3/library/typing.html)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

## Ready to Start

This document now provides a clear Phase 1 starting point for systematic MyPy improvements. Each step is designed to provide immediate value while maintaining code quality and test compatibility.

**Next Action**: Begin with Step 1.1 - Add Basic Type Annotations to `src/giflab/metrics.py`