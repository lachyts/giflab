# Multi-Engine Roll-Out Plan

_Rev 2 – updated 2025-07-22 with progress tracking_

The goal of this work stream is to replace the **stub** wrappers for ImageMagick, FFmpeg and gifski with fully-functional implementations, align the configuration / discovery layer, and add lightweight yet meaningful test coverage similar to the existing gifsicle / animately integration tests.

The project standardizes on American English spelling (e.g., "color") across all documentation.

---
## 1  Scope & success criteria

| Engine | Actions to support | Success criteria |
|--------|-------------------|------------------|
| ImageMagick (`convert`, `magick`) | • color reduction  
• frame reduction  
• lossy compression (quality / sampling) | • CLI executes without error  
• Output GIF exists  
• Palette / frame-count / size reflect requested change  
• Metadata dict includes `render_ms`, `engine`, `command`, `kilobytes` |
| FFmpeg | • color reduction (palettegen / paletteuse)  
• frame reduction (`fps=` filter)  
• lossy compression (`qscale`, sampling) | *Same as above* |
| gifski | • lossy compression (quality slider) | *Same as above* |

*Note – ImageMagick & FFmpeg color/frame reduction can be indirect (re-encode via PNG sequence) as long as the wrapper returns a valid GIF and meets the functional checks.*

---
## 2  Implementation tasks

Below is the ordered work-breakdown with completion status tracked.

| # | Task ID | Status | Description | Notes |
|---|---------|--------|-------------|-------|
| 1 | **engines-cli-design** | ✅ **DONE** | Document exact CLI invocations for each engine / action. | Examples saved to `docs/cli_recipes.md`. |
| 2 | **helpers-impl** | ✅ **DONE** | Add helper functions under `giflab.external_engines.*` that wrap each CLI and return the standard metadata dict. | Must populate `render_ms`, `engine`, `command`, `kilobytes`. |
| 3 | **wrappers-update** | ✅ **DONE** | Replace placeholder `apply()` methods in ImageMagick/FFmpeg/gifski wrappers with real calls to the helpers. | Ensure `COMBINE_GROUP` values stay consistent. |
| 4 | **config-update** | ✅ **DONE** | Extend `DEFAULT_ENGINE_CONFIG` and `system_tools.discover_tool` to auto-detect executables; allow `$GIFLAB_IMAGEMAGICK_PATH`, `$GIFLAB_FFMPEG_PATH`, `$GIFLAB_GIFSKI_PATH`, `$GIFLAB_GIFSICLE_PATH`, `$GIFLAB_FFPROBE_PATH` overrides. | Added full EngineConfig with env var support and updated system_tools.discover_tool |
| 5 | **fixtures** | ⏳ **TODO** | Add 1–2 tiny GIF fixtures for palette, frame-count and size assertions. | Store under `tests/fixtures/`. |
| 6 | **tests-integration** | ⏳ **TODO** | Create `tests/test_engine_integration_extended.py` with one functional test per *(engine × action)*. | Validate functional change + metadata. |
| 7 | **smoke-extend** | ⏳ **TODO** | Remove skips in `test_engine_smoke.py` and add functional asserts for the new engines. | Ensure existing gifsicle / Animately smoke tests remain green and runs in fast suite. |
| 8 | **ci-update** | ⏳ **TODO** | Update CI workflow / Docker image to include ImageMagick, FFmpeg, gifski so the tests pass in CI. | |
| 9 | **docs-update** | ⏳ **TODO** | Refresh README and technical docs to list the new engines, environment variables and usage examples. | |
| 10 | **cleanup-stubs** | ⏳ **TODO** | Remove obsolete stub wrappers once real implementations are merged. | |

---
## 3  CLI recipes (draft)

### 3.1  ImageMagick
```bash
# color reduction to N colors
magick input.gif +dither -colors 32 output.gif

# frame reduction – keep 50 % frames (delete every 2nd frame)
magick input.gif -coalesce -delete '1--2' -layers optimize output.gif
# (wrapper deletes every second frame; adjust ratio dynamically)

# lossy – sample + strip metadata
magick input.gif -sampling-factor 4:2:0 -strip -quality 85 output.gif
```

### 3.2  FFmpeg
```bash
# color reduction via palette
ffmpeg -i input.gif -filter_complex "fps=15,palettegen" palette.png
ffmpeg -i input.gif -i palette.png -filter_complex "fps=15,paletteuse" output.gif
# (wrapper can run two-pass transparently)

# frame reduction only
ffmpeg -i input.gif -filter_complex "fps=7.5" output.gif

# lossy (quantiser 30)
ffmpeg -i input.gif -lavfi "fps=15" -q:v 30 output.gif
```

### 3.3  gifski
```bash
# lossy quality 60 %
gifski --quality 60 -o output.gif input_frames/*.png
# (wrapper will split GIF to PNGs first; OK for small smoke tests)
```

---
## 4  Testing strategy

1. **Smoke tests (`tests/test_engine_smoke.py`)**  
   – Already in place, will gain functional assertions for the new engines once wrappers are real.

2. **Per-engine integration tests**  
   – For each engine/action: generate toy GIF ➞ run wrapper ➞ assert functional change + metadata.

3. **Fail-fast on CI**  
   – Mark integration tests `@pytest.mark.external_tools` and run them in a dedicated workflow job where the binaries are pre-installed.

### 4.1  Functional validation criteria

**Color reduction tests:**
- Input: 256-color GIF → `colors=32` → Output: ≤32 colors (measured via palette extraction)
- Metadata: `engine` field matches expected value, `render_ms > 0`, `kilobytes > 0`
- Edge case: Single-color GIF should remain stable

**Frame reduction tests:**  
- Input: 20-frame GIF → `keep_ratio=0.5` → Output: ~10 frames (±1 frame tolerance)
- Timing validation: Output duration should be ~50% of input duration
- Edge case: Single-frame GIF with `keep_ratio < 1.0` should handle gracefully

**Lossy compression tests:**
- Input/output file size: Output should be smaller than input (compression achieved)
- Quality degradation: PSNR should decrease but remain above minimum threshold (e.g., >20dB)
- Metadata completeness: All required fields present and non-zero

**Cross-engine consistency:**
- Same operation on same input should produce similar file sizes across engines (±20% tolerance)
- All engines should populate identical metadata schema
- Error handling should be consistent (same exception types)

### 4.2  Test fixtures

**Minimal fixtures (≤1KB each):**
- `simple_4frame.gif`: 4 frames, 16 colors, 64x64px - basic functionality
- `single_frame.gif`: 1 frame, 8 colors, 32x32px - edge case testing
- `many_colors.gif`: 4 frames, 256 colors, 64x64px - palette stress test

**Fixture validation:**
- All fixtures verified to be well-formed GIFs
- Known properties documented (frame count, color count, dimensions)
- Stored under `tests/fixtures/` with descriptive names

### 4.3  Performance regression detection

**Timing thresholds (smoke tests only):**
- Operations should complete within 5x the baseline gifsicle time
- No operation should exceed 30 seconds on test fixtures
- Memory usage should remain reasonable (no >100MB allocations for small inputs)

---
## 5  Milestones & sequencing

### Progress Summary
**Completed:** 4/10 tasks (40%)  
**In Progress:** 0/10 tasks  
**Remaining:** 6/10 tasks (60%)  

### Milestone Status
- ✅ **Stage 1-4:** CLI recipes documented + helper functions implemented + wrappers updated + configuration & environment variables
- ⏳ **Stage 5:** Test fixtures & integration tests (next)
- ⏳ **Stage 6:** CI pipeline updates
- ⏳ **Stage 7:** Documentation refresh

### Original Sequencing Plan
1. ✅ ImageMagick helpers & wrapper swap-in → merge.  
2. ✅ FFmpeg helpers & wrappers → merge.  
3. ✅ gifski lossy helper & wrapper → merge.  
4. ⏳ Extended smoke + integration tests.  
5. ⏳ CI and docs update.

Each step should keep the existing gifsicle / Animately tests green.

---
## 6  Open questions

* Reliable frame-drop recipe for ImageMagick without giant intermediate files?  
* Acceptable quality / performance trade-off for gifski PNG split in integration tests.

---
## 7  Error handling & risk mitigation

### 7.1  Failure modes & responses

**Missing executables:**
- All wrappers call `discover_tool().require()` early, failing fast with clear setup instructions
- CI/local development should install all engines or skip tests with `@pytest.mark.external_tools`

**Command execution failures:**
- `subprocess.run(..., check=True)` raises on non-zero exit codes
- Error messages include full stderr output for debugging
- 60-second timeout prevents infinite hangs on corrupted inputs

**Integration test stability:**
- Use small, well-formed GIF fixtures to minimize flakiness
- Validate functional changes (palette size, frame count) rather than exact byte-for-byte output
- Mark tests with `@pytest.mark.external_tools` for conditional execution

### 7.2  Rollback strategy

If any stage introduces regressions:
1. **Immediate:** Revert the problematic wrapper's `apply()` method to stub behavior
2. **Short-term:** Fix the underlying issue in `external_engines.*` helpers
3. **Long-term:** Add regression tests to prevent recurrence

### 7.3  Testing isolation

- Each engine's tests should be independent (no shared state)
- Use temporary directories for all intermediate files
- Clean up PNG frames and palette files automatically via `tempfile.TemporaryDirectory()`

---
**End of document** 