# 🎯 Cursor Thread Instructions: GifLab Stage 5 Implementation

## Context & Critical Discovery

I'm implementing **Stage 5** (`metrics.py`) for the GifLab project. Previous analysis revealed a **critical flaw** in naive SSIM comparison that invalidates quality measurements.

### The Frame Alignment Problem (CRITICAL)
When GIF compression removes frames, naive frame-by-frame comparison fails:

```
Original:   [0,1,2,3,4,5,6,7,8,9]  (10 frames)
Compressed: [0,2,4,6,8]            (5 frames - every 2nd kept)

❌ WRONG: Compare orig[1] with comp[1] → comparing frame 1 with frame 2
✅ RIGHT: Compare orig[2] with comp[1] → comparing frame 2 with frame 2
```

**Impact**: 18-53% SSIM improvement with correct alignment vs naive approach.

## Required Implementation

### 1. Multi-Metric Quality Assessment System

**Primary Function**: `calculate_comprehensive_metrics(original_path, compressed_path) -> Dict[str, float]`

**Return Dictionary Must Include**:
```python
{
    "ssim": float,                    # Traditional SSIM (0.0-1.0)
    "ms_ssim": float,                 # Multi-scale SSIM (0.0-1.0) 
    "psnr": float,                    # PSNR normalized (0.0-1.0)
    "temporal_consistency": float,     # Animation smoothness (0.0-1.0)
    "composite_quality": float,        # Weighted combination (0.0-1.0)
    "render_ms": int,                 # Processing time in milliseconds
    "kilobytes": float                # File size in KB
}
```

### 2. Frame Alignment Implementation (REQUIRED)

**Single alignment method (most robust)**:

1. **content_based** - Find most similar frames using MSE (visual matching)

### 3. Configuration System

```python
class MetricsConfig:
    SSIM_MODE: str = "comprehensive"  # fast, optimized, full, comprehensive
    SSIM_MAX_FRAMES: int = 30
    # Frame alignment: Always content-based (most robust visual matching)
    USE_COMPREHENSIVE_METRICS: bool = True
    TEMPORAL_CONSISTENCY_ENABLED: bool = True
```

### 4. Composite Quality Formula (EXACT)

```python
composite_quality = (0.30 * ssim + 
                    0.35 * ms_ssim + 
                    0.25 * psnr_normalized + 
                    0.10 * temporal_consistency)
```

### 5. Required Dependencies

```python
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# Note: MS-SSIM requires custom implementation or additional library
```

## Technical Requirements

### Performance Targets
- **Comprehensive mode**: ~25-40ms per GIF comparison
- **Max 30 frames** sampling for balance of accuracy/speed
- **Bulletproof mode**: 5 hours for 10,000 GIFs

### Error Handling Requirements
- Corrupted GIF files
- Single-frame GIFs
- Different dimensions (resize smaller to match)
- Memory constraints
- Processing timeouts

### Testing Requirements
- All edge cases covered
- Performance benchmarks
- Quality differentiation validation (should achieve 40%+ dynamic range)

## Critical Success Metrics

The implementation MUST achieve:
- ✅ **Frame alignment solved** - No comparing wrong frames
- ✅ **Multi-metric approach** - Eliminates single-metric bias
- ✅ **40%+ quality differentiation** - Excellent/poor quality separation
- ✅ **Production performance** - Sub-second processing per GIF

## Files to Create/Modify

1. **`src/giflab/metrics.py`** - Main implementation
2. **`tests/test_metrics.py`** - Comprehensive test suite  
3. **Update `src/giflab/config.py`** - Add MetricsConfig class

## Key Implementation Notes

- **Frame extraction**: Use `PIL.Image.seek(frame_index)` for GIF frames
- **Dimension handling**: Resize to smallest common dimensions
- **MS-SSIM**: May need custom implementation (5-scale pyramid)
- **Temporal consistency**: Analyze frame-to-frame difference correlation
- **Memory management**: Process frames in batches for large GIFs

## Validation Approach

Test with controlled quality samples:
- Excellent: composite_quality = 0.70-0.95
- Good: composite_quality = 0.55-0.70
- Poor: composite_quality = 0.35-0.55
- Severe: composite_quality = 0.15-0.35

## Reference Documentation

See `QUALITY_METRICS_APPROACH.md` for comprehensive technical details, including:
- Frame alignment algorithms
- Multi-metric theory
- Performance benchmarks
- Implementation architecture

## Expected CSV Output Columns

The metrics.py implementation should support adding these columns to the GifLab CSV:
- `ssim`, `ms_ssim`, `psnr`, `temporal_consistency`, `composite_quality`

**Primary metric for downstream analysis**: `composite_quality` (0.0-1.0 scale)

---

**TL;DR**: Implement bulletproof multi-metric quality assessment with smart frame alignment. The frame alignment problem is the critical blocker - solve this first, then add the multi-metric system for statistically reliable quality measurements. 