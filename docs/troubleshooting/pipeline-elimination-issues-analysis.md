# 🚨 Pipeline Elimination Root Cause Analysis - FFmpeg→PNG→Gifski Failure Chain

**Analysis Date**: 2025-07-27  
**Critical Issue**: FFmpeg dimension corruption breaking gifski PNG sequences  
**Impact**: 95.8% of failures (1,640/1,712) are gifski-related due to corrupted PNG input  

---

## 🔍 **Root Cause Identified**

### **The Failure Chain**
1. **FFmpeg processes GIF** with palette operations (`palettegen` + `paletteuse`)
2. **FFmpeg corrupts dimensions** (120×120 → 22×25 + 96 other sizes) - **THIS IS THE BUG**
3. **FFmpeg exports corrupted PNG sequence** with inconsistent frame sizes
4. **Gifski receives bad PNG frames** and fails validation (expects 80% consistent frames)
5. **Pipeline marked as failed** (95.8% failure rate)

### **Evidence**
- **Input**: `very_long_animation.gif` (120×120, 100% consistent frames)
- **After FFmpeg**: 97 different frame sizes (4% consistency rate)
- **Gifski requirement**: 80% frame consistency (20% tolerance)
- **Result**: Systematic failure - 4% << 80% required

### **Why Gifski Fails Most**
- **Gifski relies on PNG sequences** from previous pipeline steps
- **Other tools** (gifsicle, imagemagick, Animately) work directly with GIFs
- **FFmpeg corruption only affects** the PNG sequence export path to gifski

---

## 🛠️ **Investigation & Fix Strategy**

### **Stage 1: Reproduce FFmpeg Dimension Bug** ⚡ **(15 minutes)**

#### **Objective**: Confirm FFmpeg is changing frame dimensions during palette operations

#### **Tasks**:
1. **Verify original test file integrity**
   ```bash
   cd /Users/lachlants/repos/animately/giflab
   
   # Check original dimensions
   ffprobe -v quiet -select_streams v:0 -show_entries frame=width,height -of csv=p=0 test_elimination/very_long_animation.gif | head -10
   # Expected: All frames should be 120×120
   ```

2. **Test FFmpeg palette operations in isolation**
   ```bash
   # Copy test file
   cp test_elimination/very_long_animation.gif debug_test.gif
   
   # Run exact FFmpeg commands from our codebase
   ffmpeg -y -v error -i debug_test.gif -filter_complex "palettegen" debug_palette.png
   ffmpeg -y -v error -i debug_test.gif -i debug_palette.png -filter_complex "paletteuse" debug_output.gif
   
   # Check output dimensions (THIS SHOULD REVEAL THE BUG)
   ffprobe -v quiet -select_streams v:0 -show_entries frame=width,height -of csv=p=0 debug_output.gif | head -10
   # BUG: Shows 22×25 instead of 120×120
   ```

3. **Test FFmpeg PNG sequence export**
   ```bash
   # Test PNG sequence export (what gifski receives)
   mkdir debug_png_frames
   ffmpeg -y -v error -i debug_output.gif debug_png_frames/frame_%04d.png
   
   # Check individual PNG frame dimensions
   identify debug_png_frames/frame_0001.png debug_png_frames/frame_0002.png debug_png_frames/frame_0003.png
   # Should show inconsistent sizes (22×25, etc.)
   ```

#### **Expected Results**:
- ✅ Original GIF: 120×120 (all frames consistent)
- ❌ After FFmpeg palette: 22×25 + mixed sizes (bug confirmed)
- ❌ PNG export: Inconsistent frame sizes (gifski input corrupted)

---

### **Stage 2: Isolate FFmpeg Command Issue** 🔍 **(30 minutes)**

#### **Objective**: Determine exactly which FFmpeg command/parameter is causing dimension changes

#### **Tasks**:
1. **Test FFmpeg version and configuration**
   ```bash
   ffmpeg -version
   # Document version - check for known FFmpeg bugs
   
   # Test with verbose output
   ffmpeg -y -v debug -i debug_test.gif -i debug_palette.png -filter_complex "paletteuse" -f null -
   # Look for warnings about scaling/resizing
   ```

2. **Test alternative FFmpeg syntaxes**
   ```bash
   # Alternative 1: Use -lavfi instead of -filter_complex
   ffmpeg -y -v error -i debug_test.gif -i debug_palette.png -lavfi "paletteuse" alt1_result.gif
   
   # Alternative 2: Explicit dimension preservation
   ffmpeg -y -v error -i debug_test.gif -i debug_palette.png -filter_complex "[0:v][1:v]paletteuse" alt2_result.gif
   
   # Alternative 3: Use -vf for palette generation
   ffmpeg -y -v error -i debug_test.gif -vf "palettegen" alt_palette.png
   ffmpeg -y -v error -i debug_test.gif -i alt_palette.png -lavfi "paletteuse" alt3_result.gif
   
   # Check dimensions of all outputs
   for file in alt1_result.gif alt2_result.gif alt3_result.gif; do
       echo "=== $file ==="
       ffprobe -v quiet -select_streams v:0 -show_entries frame=width,height -of csv=p=0 $file | head -3
   done
   ```

3. **Test with different test files**
   ```bash
   # Test with other GIFs to see if it's content-specific
   cp test_elimination/many_colors.gif test2.gif
   ffmpeg -y -v error -i test2.gif -filter_complex "palettegen" test2_palette.png
   ffmpeg -y -v error -i test2.gif -i test2_palette.png -filter_complex "paletteuse" test2_output.gif
   
   ffprobe -v quiet -select_streams v:0 -show_entries frame=width,height -of csv=p=0 test2_output.gif | head -5
   # Check if dimension corruption is systematic
   ```

#### **Expected Outcome**: 
- Find the specific FFmpeg command/syntax causing dimension corruption
- Determine if it's FFmpeg version-specific or command-specific

---

### **Stage 3: Implement FFmpeg Fix** 🔧 **(30 minutes)**

#### **Objective**: Fix FFmpeg commands to preserve dimensions correctly

#### **Tasks**:
1. **Update FFmpeg color reduction functions**
   ```bash
   # Edit src/giflab/external_engines/ffmpeg.py
   # Based on Stage 2 findings, implement the working FFmpeg syntax
   ```

2. **Test fix with working FFmpeg syntax**
   ```python
   # Example fix (adjust based on Stage 2 results):
   # In color_reduce() function, change:
   use_cmd = [
       ffmpeg, "-y", "-v", "error",
       "-i", str(input_path),
       "-i", str(palette_path),
       "-filter_complex", "[0:v][1:v]paletteuse",  # Explicit input mapping
       str(output_path),
   ]
   ```

3. **Update FFmpeg enhanced functions**
   ```bash
   # Edit src/giflab/external_engines/ffmpeg_enhanced.py
   # Apply same fix to color_reduce_with_dithering()
   ```

#### **Expected Outcome**:
- FFmpeg preserves 120×120 → 120×120 dimensions
- PNG sequence export creates consistent frame sizes

---

### **Stage 4: Test Pipeline Chain Fix** ✅ **(20 minutes)**

#### **Objective**: Verify the entire pipeline chain works with fixed FFmpeg

#### **Tasks**:
1. **Test minimal failing pipeline**
   ```bash
   # Test the top failing combination: none-frame + ffmpeg-color + gifski-lossy
   poetry run python -c "
   from giflab.dynamic_pipeline import generate_all_pipelines
   from giflab.pipeline_elimination import PipelineEliminator
   
   eliminator = PipelineEliminator('debug_fix_test')
   pipelines = [p for p in generate_all_pipelines() 
                if 'none-frame' in p.identifier() and 'ffmpeg-color' in p.identifier() and 'gifski-lossy' in p.identifier()]
   
   print(f'Testing {len(pipelines)} previously failing pipelines...')
   # Should now succeed with consistent 120×120 dimensions
   "
   ```

2. **Test PNG sequence generation**
   ```bash
   # Test that FFmpeg now creates consistent PNG sequences
   mkdir test_png_fixed
   ffmpeg -y -v error -i debug_output.gif test_png_fixed/frame_%04d.png
   
   # Verify all frames have same dimensions
   identify test_png_fixed/frame_*.png | awk '{print $3}' | sort | uniq -c
   # Should show all frames have same size
   ```

3. **Test gifski with fixed PNG sequence**
   ```bash
   # Test gifski directly with fixed PNG frames
   poetry run python -c "
   from giflab.external_engines.gifski import lossy_compress
   from pathlib import Path
   
   result = lossy_compress(
       Path('debug_test.gif'),
       Path('gifski_test_output.gif'),
       png_sequence_dir=Path('test_png_fixed')
   )
   print('Gifski result:', result)
   # Should succeed without dimension validation errors
   "
   ```

#### **Expected Outcome**:
- Pipeline succeeds end-to-end
- Gifski processes PNG sequence without validation errors
- Failure rate drops dramatically

---

### **Stage 5: Verify Non-Gifski Pipelines** 🔍 **(10 minutes)**

#### **Objective**: Confirm other pipeline endings work correctly (they should)

#### **Tasks**:
1. **Test gifsicle pipelines**
   ```bash
   poetry run python -c "
   pipelines = [p for p in generate_all_pipelines() if 'gifsicle-lossy' in p.identifier()]
   print(f'Testing {len(pipelines)} gifsicle pipelines (should work)...')
   # Test a few representative ones
   "
   ```

2. **Test ImageMagick pipelines**
   ```bash
   poetry run python -c "
   pipelines = [p for p in generate_all_pipelines() if 'imagemagick-lossy' in p.identifier()]
   print(f'Testing {len(pipelines)} imagemagick pipelines (should work)...')
   "
   ```

#### **Expected Outcome**:
- Non-gifski pipelines work normally (they don't use PNG sequences)
- Confirms the issue is specific to the FFmpeg→PNG→gifski chain

---

### **Stage 6: Database Verification** 📊 **(15 minutes)**

#### **Objective**: Verify the fix reduces failure rate in actual pipeline elimination

#### **Tasks**:
1. **Query current failure patterns**
   ```bash
   # Check current gifski failure rate
   poetry run python -m giflab debug-failures --summary
   
   # Focus on gifski failures
   poetry run python -m giflab debug-failures --error-type gifski
   ```

2. **Run limited test with fix**
   ```bash
   # Test with small sample to verify fix
   poetry run python -m giflab eliminate-pipelines \
     --sampling-strategy quick \
     --max-pipelines 100 \
     -o test_fix_results
   ```

3. **Compare failure rates**
   ```bash
   # Check new failure distribution
   sqlite3 test_fix_results/pipeline_results_cache.db "
   SELECT error_type, COUNT(*) as count 
   FROM pipeline_failures 
   GROUP BY error_type 
   ORDER BY count DESC;"
   
   # Target: Reduce gifski failures from 95.8% to <5%
   ```

#### **Expected Outcome**:
- Dramatic reduction in gifski failures (from 1,640 to <50)
- Overall failure rate drops from 67% to <10%

---

### **Stage 7: Production Update** 🚀 **(15 minutes)**

#### **Objective**: Apply the fix to production and verify full pipeline elimination

#### **Tasks**:
1. **Commit the FFmpeg fixes**
   ```bash
   git add src/giflab/external_engines/ffmpeg.py src/giflab/external_engines/ffmpeg_enhanced.py
   git commit -m "Fix FFmpeg dimension corruption in palette operations
   
   - Preserve frame dimensions during palettegen/paletteuse operations
   - Fixes 95.8% of pipeline elimination failures
   - Ensures consistent PNG sequences for gifski processing"
   ```

2. **Clear old failures and re-run**
   ```bash
   # Clear fixed failures from database
   poetry run python -m giflab debug-failures --clear-fixed
   
   # Re-run full pipeline elimination
   poetry run python -m giflab eliminate-pipelines \
     --sampling-strategy comprehensive \
     -o final_results
   ```

3. **Verify final results**
   ```bash
   # Monitor results
   poetry run python monitor_elimination_enhanced.py
   
   # Expected: High success rate, minimal gifski failures
   ```

#### **Expected Outcome**:
- Pipeline elimination completes successfully (>90% completion rate)
- Gifski pipelines work correctly
- Valid elimination results for all pipeline combinations

---

## 📋 **Quick Reference Commands**

### **Reproduce the Bug**
```bash
cp test_elimination/very_long_animation.gif debug_test.gif
ffmpeg -y -v error -i debug_test.gif -filter_complex "palettegen" debug_palette.png
ffmpeg -y -v error -i debug_test.gif -i debug_palette.png -filter_complex "paletteuse" debug_output.gif
ffprobe -v quiet -select_streams v:0 -show_entries frame=width,height -of csv=p=0 debug_output.gif | head -10
```

### **Check Failure Patterns**
```bash
poetry run python -m giflab debug-failures --summary
poetry run python -m giflab debug-failures --error-type gifski
```

### **Test Specific Pipeline**
```bash
poetry run python -c "
from giflab.dynamic_pipeline import generate_all_pipelines
pipelines = [p for p in generate_all_pipelines() if 'ffmpeg-color' in p.identifier() and 'gifski-lossy' in p.identifier()]
print(f'Found {len(pipelines)} FFmpeg→gifski pipelines to test')
"
```

---

## 🎯 **Success Criteria**

- [ ] **Stage 1**: Confirm FFmpeg changes 120×120 → 22×25 (bug reproduction)
- [ ] **Stage 2**: Identify specific FFmpeg command causing dimension corruption  
- [ ] **Stage 3**: Fix FFmpeg commands to preserve dimensions
- [ ] **Stage 4**: Verify pipeline chain works end-to-end
- [ ] **Stage 5**: Confirm non-gifski pipelines still work
- [ ] **Stage 6**: Database shows <5% gifski failure rate
- [ ] **Stage 7**: Full pipeline elimination completes successfully

**Target Outcome**: Reduce failure rate from 95.8% to <5% by fixing FFmpeg dimension preservation 