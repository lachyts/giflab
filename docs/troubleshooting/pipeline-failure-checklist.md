# 🔥 Pipeline Failure Quick Action Checklist

> **Primary Issue**: gifski frame size mismatches in `animately-frame + ffmpeg-color + gifski-lossy` pipelines

## ⚡ **Immediate Actions (Start Here)**

### 1. **Reproduce Minimal Failure** (15 minutes)
- [ ] Copy a simple test GIF: `cp test_elimination/few_colors.gif debug_test.gif`
- [ ] Run single problematic pipeline:
  ```bash
  poetry run python -c "
  from giflab.dynamic_pipeline import generate_all_pipelines
  from giflab.pipeline_elimination import PipelineEliminator
  
  # Test the specific failing combination
  eliminator = PipelineEliminator('debug_output')
  pipelines = [p for p in generate_all_pipelines() if 'animately-frame' in p.identifier() and 'ffmpeg-color' in p.identifier() and 'gifski-lossy' in p.identifier()]
  print(f'Testing {len(pipelines)} problematic pipelines...')
  "
  ```
- [ ] Capture exact error message and frame dimensions
- [ ] Document which specific frame number and size triggers the error

### 2. **Test Individual Tools** (20 minutes)
- [ ] Test animately-frame alone:
  ```bash
  # Check if animately-frame produces consistent frame sizes
  poetry run python -c "
  from giflab.external_engines.common import Engine
  # Test animately frame processing
  "
  ```
- [ ] Test ffmpeg-color alone:
  ```bash
  # Check if ffmpeg-color maintains frame dimensions
  ffmpeg -i debug_test.gif -vf palettegen -y palette.png
  ffmpeg -i debug_test.gif -i palette.png -lavfi paletteuse debug_ffmpeg.gif
  ```
- [ ] Test gifski alone:
  ```bash
  # Check gifski requirements and behavior
  gifski --help | grep -i size
  ```

### 3. **Dimension Analysis** (10 minutes)
- [ ] Check original GIF frame info:
  ```bash
  ffprobe -v quiet -select_streams v:0 -show_entries frame=width,height -of csv=p=0 debug_test.gif | head -10
  ```
- [ ] Check for frame dimension variations in original
- [ ] Document any dimension inconsistencies

## 🔍 **Investigation Phase** (Next 30-60 minutes)

### 4. **Test Alternative Pipelines**
- [ ] Test working alternative: `animately-frame + ffmpeg-color + gifsicle-lossy`
  ```bash
  poetry run python -m giflab experiment --matrix --gifs 1 --max-pipelines 10
  # Filter for working combinations only
  ```
- [ ] Test: `animately-frame + animately-color + gifski-lossy`
- [ ] Test: `gifsicle-frame + ffmpeg-color + gifski-lossy`
- [ ] Record which combinations work vs. fail

### 5. **Content Type Testing**
Test same problematic pipeline on different content:
- [ ] Simple content: `few_colors.gif` → Result: ___________
- [ ] Complex content: `many_colors.gif` → Result: ___________  
- [ ] Animation: `animation_heavy.gif` → Result: ___________
- [ ] Gradient: `smooth_gradient.gif` → Result: ___________

Record pattern: Does content type affect failure?

### 6. **Parameter Variation Testing**
- [ ] Test different color counts (8, 16, 32, 64, 256)
- [ ] Test different lossy levels (0, 40, 120)  
- [ ] Test different frame ratios (0.5, 0.7, 1.0)
- [ ] Record which parameters trigger/avoid failures

## 🛠️ **Solution Implementation** (Next 1-2 hours)

### 7. **Quick Fixes to Test**
- [ ] **Option A**: Exclude problematic pipelines
  ```python
  # In pipeline generation, filter out known bad combinations
  pipelines = [p for p in all_pipelines if not (
      'animately-frame' in p.identifier() and 
      'ffmpeg-color' in p.identifier() and 
      'gifski-lossy' in p.identifier()
  )]
  ```

- [ ] **Option B**: Add dimension validation
  ```python
  # Add frame size check before gifski step
  def validate_frame_dimensions(gif_path):
      # Check all frames have same dimensions
      # Return False if inconsistent
  ```

- [ ] **Option C**: Use alternative tool chains
  ```python
  # Replace gifski with gifsicle for lossy compression
  # Keep same frame/color processing, change only lossy step
  ```

### 8. **Implement Best Solution**
Based on testing results, implement the solution that:
- [ ] Maintains compression quality
- [ ] Preserves processing speed  
- [ ] Works across all content types
- [ ] Has minimal code changes

## ✅ **Validation Phase** (30 minutes)

### 9. **Test Fix with Full Set**
- [ ] Run elimination testing with fixes:
  ```bash
  poetry run python -m giflab eliminate-pipelines --sampling-strategy quick --max-pipelines 100 -o test_fix_results
  ```
- [ ] Check failure rate improvement
- [ ] Verify quality metrics remain good
- [ ] Confirm no new failure patterns emerge

### 10. **Update Documentation**
- [ ] Update pipeline compatibility matrix
- [ ] Document any tool limitations discovered
- [ ] Add warnings to problematic combinations
- [ ] Update troubleshooting guide with findings

## 📊 **Results Tracking**

### Failure Analysis Results:
- **Root Cause**: ________________________________
- **Working Alternatives**: ________________________
- **Fix Implemented**: ____________________________
- **Failure Rate Before**: ___% → **After**: ___%

### Content Type Impact:
- **Gradients**: ⚪ Fixed / ❌ Still failing
- **Geometric**: ⚪ Fixed / ❌ Still failing  
- **High-color**: ⚪ Fixed / ❌ Still failing
- **Animation**: ⚪ Fixed / ❌ Still failing

### Performance Impact:
- **Quality Change**: +/- ___% SSIM
- **Speed Change**: +/- ___% processing time
- **Compression Change**: +/- ___% file size

## 🚀 **Final Steps**

### 11. **Production Update**
- [ ] Apply fixes to main pipeline generation
- [ ] Update configuration defaults
- [ ] Add regression tests for fixed issues
- [ ] Re-run full elimination analysis

### 12. **Documentation Update**
- [ ] Update main README.md with any tool limitations
- [ ] Add troubleshooting section referencing this checklist
- [ ] Document any pipeline changes in release notes

---

## 🆘 **If Stuck / Need Help**

**Stop and escalate if:**
- Multiple alternative pipelines show same issues
- Root cause points to fundamental tool incompatibility  
- Fix requires major architectural changes
- Quality/performance impact is unacceptable

**Quick debugging commands:**
```bash
# Get tool versions
poetry run python -c "from giflab.system_tools import get_available_tools; print(get_available_tools())"

# Check specific tool output
poetry run python -c "
from giflab.external_engines.gifski import GifskiEngine
engine = GifskiEngine()
print(f'Gifski available: {engine.is_available()}')
"

# Test with minimal GIF
poetry run python -m giflab view-failures test_failure_logging/ --error-type gifski --limit 3 --detailed
```

---

*Estimated total time: 2-4 hours*  
*Priority: High - blocking elimination testing* 